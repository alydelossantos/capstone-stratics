/*==================== SHOW NAVBAR ====================*/
const showMenu = (headerToggle, sbarId) =>{
    const toggleBtn = document.getElementById(headerToggle),
    nav = document.getElementById(sbarId)
    
    // Validate that variables exist
    if(headerToggle && sbarId){
        toggleBtn.addEventListener('click', ()=>{
            // We add the show-menu class to the div tag with the nav__menu class
            nav.classList.toggle('show-menu')
            // change icon
            toggleBtn.classList.toggle('bx-x')
        })
    }
}
showMenu('header-toggle','sbar')

/*==================== LINK ACTIVE ====================*/
const linkColor = document.querySelectorAll('.sb__link')

function colorLink(){
    linkColor.forEach(l => l.classList.remove('active'))
    this.classList.add('active')
}

linkColor.forEach(l => l.addEventListener('click', colorLink))
